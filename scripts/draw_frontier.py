import argparse
import moplayground as mop
from moplayground.utils.config import MOConfig
from matplotlib import pyplot as plt
import jax

parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="YAML config of the run to evaluate")
args = parser.parse_args()

cfg = mop.MOConfig(args.config_path)
env, env_config = mop.create_environment(
    env_name     = cfg.env_name,
    env_params   = cfg.env_params,
    backend      = 'jnp',
    # gaitlib_path = cfg.gaitlib_path,
    # manual_speed = True,
    # idealistic   = True
)
get_fronts = {
    'morlax': mop.get_morlax_fronts,
    'amor':   mop.get_amor_fronts,
}[cfg.algorithm(cfg)]
num_objectives = cfg.num_objectives
rewards_over_iters, tradeoffs = get_fronts(
    run_dirs        = cfg.run_dir,
    network_params  = cfg.network_params,
    num_objectives  = num_objectives,
    rng             = jax.random.PRNGKey(95),
    env             = env,
    N_STEPS         = 500,
    NUM_ENVS        = 1024,
    save_results    = True,
    only_final      = True
)
print(rewards_over_iters.shape)
nd_idx = mop.get_nondominated(rewards_over_iters[-1], epsilon=10)

if(num_objectives == 3):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
elif(num_objectives == 2):
    fig, ax = plt.subplots()
else:
    raise ValueError("Can only plot 2 or 3 objective pareto frontiers")


ax = mop.plot_pareto(
    ax          = ax,
    pareto      = rewards_over_iters[-1],
    colors      = mop.default_coloring(tradeoffs[-1]),
    objective   = cfg.objective_labels,
)

plt.show()
# fig.savefig(f'output/plots/{config.env_name(cfg)}_frontier.pdf')
